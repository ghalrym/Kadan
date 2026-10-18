#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>

#include "kadan/image_aten_plan.hpp"
#include "kadan/image_batch_memory.hpp"
#include "kadan/image_batch_output.hpp"
#include "kadan/image_edit.hpp"
#include "kadan/image_generation.hpp"
#include "kadan/image_upload.hpp"
#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
Bytes resident_bytes() {
    std::ifstream stat("/proc/self/statm");
    Bytes pages = 0, resident = 0;
    if (!(stat >> pages >> resident))
        throw std::runtime_error("image_host_memory_probe");
    return resident * static_cast<Bytes>(sysconf(_SC_PAGESIZE));
}

// RSS is sampled only around a batch, including its preparation and output
// buffers. Process lifetime ru_maxrss cannot measure a later batch's increment.
class BatchMonitor {
public:
    const Bytes baseline = resident_bytes();

    explicit BatchMonitor(std::function<void()> update_cancellation)
        : peak_(baseline), sampler_([this, update_cancellation] {
              std::unique_lock lock(mutex_);
              while (!done_) {
                  update_cancellation();
                  try {
                      peak_ = std::max(peak_, resident_bytes());
                  } catch (...) {
                      failed_ = true;
                  }
                  changed_.wait_for(lock, std::chrono::milliseconds(10), [&] {
                      return done_;
                  });
              }
          }) {
    }

    ~BatchMonitor() {
        stop();
    }

    Bytes finish() {
        stop();
        if (failed_)
            throw std::runtime_error("image_host_memory_probe");
        return std::max(peak_, resident_bytes());
    }

private:
    void stop() {
        {
            std::lock_guard lock(mutex_);
            done_ = true;
        }
        changed_.notify_one();
        if (sampler_.joinable())
            sampler_.join();
    }

    Bytes peak_;
    bool done_ = false, failed_ = false;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread sampler_;
};

image::ImageRequest native_request(const ImageRequest& input) {
    return {
        input.prompt,
        input.width,
        input.height,
        input.steps,
        *input.seed,
        {},
        {},
        input.strength,
        input.reference,
        input.source_width,
        input.source_height
    };
}

struct BatchShape {
    unsigned width, height, steps;
    bool reference;
    std::size_t source_width, source_height;
    auto operator<=>(const BatchShape&) const = default;
};

BatchShape batch_shape(const ImageRequest& input) {
    return {
        input.width,
        input.height,
        input.steps,
        input.reference,
        input.source_width,
        input.source_height
    };
}
}  // namespace

class ImageEngine final : public InferenceEngine {
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    RequestProgress request_progress_;
    InferenceOwner::Probe probe_;
    std::atomic_bool cancel_{false};
    std::function<void()> update_cancel_;
    std::unique_ptr<WorkerCompute> compute_;
    std::unique_ptr<image::Generator> model_;
    std::unique_ptr<HostAllocation> request_memory_;
    std::unique_ptr<BatchMonitor> host_peak_;
    std::vector<float> source_, mask_;
    std::vector<image::ImageRequest> images_;

    struct ImageOwner {
        std::size_t request, image;
    };

    std::vector<ImageOwner> image_owners_;
    std::vector<ImageBatchOutput> outputs_;
    std::map<BatchShape, ImageBatchMemory> measured_;
    std::size_t capacity_ = 1;

    void release_request() {
        host_peak_.reset();
        update_cancel_ = {};
        images_.clear();
        image_owners_.clear();
        outputs_.clear();
        std::vector<float>().swap(source_);
        std::vector<float>().swap(mask_);
        request_memory_.reset();
    }

public:
    ImageEngine(
        std::shared_ptr<Resources> r,
        Progress p,
        RequestProgress rp,
        InferenceOwner::Probe probe
    )
        : resources_(std::move(r)), progress_(std::move(p)),
          request_progress_(std::move(rp)), probe_(std::move(probe)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        if (available[0] < 16ULL * 1024 * 1024 * 1024)
            throw AdmissionWait("image_host_headroom");
        compute_ = std::make_unique<WorkerCompute>(
            media_plan(available), Workload::image, resources_
        );
        model_ = std::make_unique<image::Generator>(resources_, compute_->devices);
        model_->load(request.configuration.checkpoint.string(), cancel, progress_);
    }

    std::size_t batch_capacity(
        const InferenceRequest& request, const Footprint& available
    ) override {
        capacity_ = 1;
        if (request.operation != Operation::generate
            || !compatible_images(request, request))
            return capacity_;
        const auto& input = std::get<ImageRequest>(request.input);
        const std::atomic_bool cancel{false};
        const auto planned =
            model_->admissible_batch_footprints(native_request(input), 32, cancel);
        auto reusable = available;
        const auto cached = model_->device_cache_memory().allocated;
        const auto snapshot = resources_->snapshot();
        for (std::size_t device = 1; device < reusable.size(); ++device)
            reusable[device] += std::min(
                cached.at(device),
                snapshot.capacity[device] > reusable[device]
                    ? snapshot.capacity[device] - reusable[device]
                    : 0
            );
        // This is a batch-size projection, not admission credit. Cached GPU
        // payloads already belong to this model; Stage must reuse or physically
        // evict them and re-probe before admitting the selected execution plan.
        capacity_ = std::max<std::size_t>(
            1, measured_[batch_shape(input)].capacity(reusable, planned)
        );
        return capacity_;
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        const InferenceBatchItem item{&request, &cancel};
        prepare_batch(std::span(&item, 1), available);
    }

    void prepare_batch(
        std::span<const InferenceBatchItem> batch, const Footprint& available
    ) override {
        release_request();
        if (batch[0].request->operation == Operation::load)
            return;
        std::vector<const std::atomic_bool*> cancellations;
        for (const auto& item : batch)
            cancellations.push_back(item.cancel);
        update_cancel_ = [this, cancellations] {
            cancel_.store(
                std::all_of(
                    cancellations.begin(),
                    cancellations.end(),
                    [](const auto* cancelled) { return cancelled->load(); }
                )
            );
        };
        update_cancel_();
        host_peak_ = std::make_unique<BatchMonitor>(update_cancel_);
        const auto& first = std::get<ImageRequest>(batch[0].request->input);
        Bytes bytes = 8 * 1024 * 1024;
        for (const auto& item : batch) {
            const auto& input = std::get<ImageRequest>(item.request->input);
            require(
                input.count == 1 || input.count == 2 || input.count == 4, "image_count"
            );
            require(
                batch.size() == 1
                    || compatible_images(*batch[0].request, *item.request),
                "incompatible_image_batch"
            );
            bytes +=
                (64ULL * input.height * input.width + 8192ULL * 4096 * sizeof(float))
                * input.count;
        }
        if (first.source)
            bytes += 32ULL * first.source_height * first.source_width;
        if (first.reference) {
            const auto points = first.source_width / 16 * (first.source_height / 16);
            bytes += image::aten_reference_host_bytes(points, points / 4, 0);
        }
        if (bytes > available[0])
            throw AdmissionWait("image_request_headroom");
        request_memory_ =
            std::make_unique<HostAllocation>(resources_, Workload::image, bytes);
        if (first.source) {
            const auto pixels = first.source_width * first.source_height;
            std::vector<std::uint8_t> rgba(pixels * 4);
            source_.resize(pixels * 4);
            decode_image_upload(*first.source, rgba, source_);
            if (first.edit_mask) {
                mask_.resize(pixels);
                decode_image_mask(*first.edit_mask, rgba, mask_);
            }
        }
        for (std::size_t index = 0; index < batch.size(); ++index) {
            const auto& input = std::get<ImageRequest>(batch[index].request->input);
            outputs_.push_back({input.count, {}, nullptr});
            auto native = native_request(input);
            native.source = source_;
            native.mask = mask_;
            try {
                image::Generator::validate(native);
                if (!batch[index].cancel->load())
                    model_->prepare(native, cancel_);
            } catch (const ResourceExhausted&) {
                throw;
            } catch (const RequestError& error) {
                outputs_.back().failure = {
                    {"error", error.what()}, {"status_code", error.status}
                };
                continue;
            } catch (const std::exception& error) {
                outputs_.back().failure = error.what();
                continue;
            }
            for (std::size_t image = 0; image < input.count; ++image) {
                native.seed = *input.seed + image;
                images_.push_back(native);
                image_owners_.push_back({index, image});
            }
        }
        while (!cancel_.load() && !images_.empty()) {
            try {
                model_->prepare_batch(
                    std::span(images_).first(std::min(capacity_, images_.size())),
                    cancel_
                );
                break;
            } catch (const ResourceExhausted&) {
                if (capacity_ == 1)
                    throw;
                --capacity_;
            }
        }
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        const InferenceBatchItem item{&request, &cancel};
        auto result = execute_batch(std::span(&item, 1)).front();
        if (!result.failure.is_null()) {
            if (result.failure.is_object())
                throw RequestError(
                    result.failure.at("error"), result.failure.at("status_code")
                );
            throw std::runtime_error(result.failure.get<std::string>());
        }
        return result.result;
    }

    std::vector<InferenceBatchResult>
    execute_batch(std::span<const InferenceBatchItem> batch) override {
        std::vector<InferenceBatchResult> results(batch.size());
        if (batch[0].request->operation == Operation::load) {
            results[0].result = Json::object();
            return results;
        }
        try {
            for (std::size_t start = 0; start < images_.size();) {
                const auto chunk_start = start;
                std::vector<image::ImageRequest> requests;
                std::vector<ImageOwner> owners;
                while (start < images_.size() && requests.size() < capacity_) {
                    const auto owner = image_owners_[start];
                    if (!batch[owner.request].cancel->load()
                        && outputs_[owner.request].failure.is_null()) {
                        requests.push_back(images_[start]);
                        owners.push_back(owner);
                    }
                    ++start;
                }
                if (requests.empty())
                    continue;
                if (!host_peak_)
                    host_peak_ = std::make_unique<BatchMonitor>(update_cancel_);
                const auto began = std::chrono::steady_clock::now();
                std::vector<std::vector<float>> rgba(requests.size());
                std::vector<std::span<float>> output;
                for (std::size_t index = 0; index < requests.size(); ++index) {
                    rgba[index].resize(
                        requests[index].height * requests[index].width * 4
                    );
                    output.push_back(rgba[index]);
                    const auto owner = owners[index];
                    if (!batch[owner.request].cancel->load())
                        request_progress_(
                            batch[owner.request].request->id,
                            "image_started",
                            owner.image + 1
                        );
                }
                update_cancel_();
                if (cancel_.load())
                    continue;
                try {
                    model_->prepare_batch(requests, cancel_);
                } catch (const ResourceExhausted&) {
                    if (capacity_ == 1)
                        throw;
                    --capacity_;
                    start = chunk_start;
                    continue;
                }
                model_->generate_batch(
                    requests,
                    output,
                    cancel_,
                    [&](std::size_t index, const char* phase, std::size_t value) {
                        update_cancel_();
                        const auto owner = owners.at(index);
                        if (!batch[owner.request].cancel->load())
                            request_progress_(
                                batch[owner.request].request->id, phase, value
                            );
                    }
                );
                for (std::size_t index = 0; index < requests.size(); ++index) {
                    const auto owner = owners[index];
                    const auto& item = batch[owner.request];
                    if (item.cancel->load())
                        continue;
                    try {
                        auto path = item.request->workspace
                                    / ("image-" + std::to_string(owner.image) + ".png");
                        image::publish_png(
                            path.string(),
                            rgba[index],
                            requests[index].height,
                            requests[index].width,
                            *item.cancel
                        );
                        outputs_[owner.request].paths.push_back(path);
                    } catch (const std::exception& error) {
                        outputs_[owner.request].failure = error.what();
                    }
                }
                const auto host_peak = host_peak_->finish();
                const auto& measurement = model_->last_batch_measurement();
                Footprint incremental(resources_->snapshot().capacity.size());
                incremental[0] = host_peak > host_peak_->baseline
                                     ? host_peak - host_peak_->baseline
                                     : 0;
                // A reused malloc arena may hide the next batch's RSS growth. Keep a
                // floor for the actual owned output buffers as well as prior peaks.
                incremental[0] = std::max<Bytes>(
                    incremental[0],
                    requests.size() * 64ULL * requests[0].width * requests[0].height
                );
                for (std::size_t device = 1; device < incremental.size(); ++device) {
                    const auto allocated = measurement.peak_allocated.at(device);
                    const auto reserved = measurement.peak_reserved.at(device);
                    incremental[device] = std::max(
                        allocated > measurement.baseline_allocated.at(device)
                            ? allocated - measurement.baseline_allocated[device]
                            : 0,
                        reserved > measurement.baseline_reserved.at(device)
                            ? reserved - measurement.baseline_reserved[device]
                            : 0
                    );
                }
                const auto& input =
                    std::get<ImageRequest>(batch[owners[0].request].request->input);
                auto per_image = measurement.additional_image;
                per_image[0] = 64ULL * requests[0].width * requests[0].height
                               + 8192ULL * 4096 * sizeof(float);
                measured_[batch_shape(input)].observe(
                    requests.size(), incremental, per_image
                );
                const Json telemetry{
                    {"event", "image_batch"},
                    {"images", requests.size()},
                    {"seconds",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - began
                     )
                         .count()},
                    {"host_baseline_bytes", host_peak_->baseline},
                    {"host_peak_bytes", host_peak},
                    {"device_peak_allocated_bytes", measurement.peak_allocated},
                    {"device_peak_reserved_bytes", measurement.peak_reserved},
                    {"incremental_peak_bytes", incremental},
                    {"additional_image_bytes", per_image}
                };
                std::clog << "KADAN_ATEN_IMAGE_BATCH " << telemetry.dump() << '\n';
                host_peak_.reset();
                if (start < images_.size() && !cancel_.load()) {
                    auto available = probe_();
                    resources_->refresh_available(available);
                    batch_capacity(*batch[owners[0].request].request, available);
                }
            }
        } catch (const RequestError& error) {
            for (auto& output : outputs_)
                output.fail_if_incomplete(
                    {{"error", error.what()}, {"status_code", error.status}}
                );
        } catch (const std::exception& error) {
            for (auto& output : outputs_)
                output.fail_if_incomplete(error.what());
        }
        compute_->idle(model_->device_cache_memory().reserved);
        for (std::size_t index = 0; index < batch.size(); ++index) {
            results[index].failure = outputs_[index].failure;
            if (batch[index].cancel->load() || !results[index].failure.is_null())
                continue;
            try {
                results[index].result =
                    present_image(*batch[index].request, outputs_[index].paths);
            } catch (const std::exception& error) {
                results[index].failure = error.what();
            }
        }
        release_request();
        return results;
    }

    void close() override {
        release_request();
        if (model_)
            model_->unload();
        model_.reset();
        if (compute_)
            compute_->park();
        compute_.reset();
    }
};

std::unique_ptr<InferenceEngine> image_engine(
    std::shared_ptr<Resources> r,
    Progress p,
    RequestProgress rp,
    InferenceOwner::Probe probe
) {
    return std::make_unique<ImageEngine>(
        std::move(r), std::move(p), std::move(rp), std::move(probe)
    );
}
}  // namespace kadan::serving
