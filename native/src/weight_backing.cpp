#include "kadan/weight_backing.hpp"
#include <algorithm>
#include <limits>

namespace kadan::serving {
namespace {
void require(bool value, const char* error) { if (!value) throw std::runtime_error(error); }
void count(Bytes& target, Bytes value) { target += std::min(value, std::numeric_limits<Bytes>::max()-target); }
void cancel(const std::atomic_bool* flag) {
    require(!flag || !flag->load(std::memory_order_relaxed), "weight_cancelled");
}
struct Busy {
    bool& flag;
    explicit Busy(bool& f) : flag(f) { require(!flag, "weight_reentrant"); flag = true; }
    ~Busy() { flag = false; }
};
}
WeightBacking::WeightBacking(std::shared_ptr<Resources> resources, Bytes ram_limit,
                             Bytes cold_limit, std::size_t chunk_limit, std::size_t entry_limit, bool aggregate)
    : resources_(std::move(resources)), ram_limit_(ram_limit), cold_limit_(cold_limit),
      chunk_limit_(chunk_limit), entry_limit_(entry_limit) {
    require(resources_ && chunk_limit && chunk_limit <= 32 * 1024 * 1024 &&
            entry_limit && entry_limit <= model_entry_limit, "weight_limits");
    require(ram_limit <= resources_->snapshot().capacity[0], "weight_ram_limit");
    if (aggregate && ram_limit) { aggregate_=resources_->reserve(Workload::llm,host(ram_limit)); resources_->loaded(aggregate_); }
}
WeightBacking::~WeightBacking() {
    // An unacknowledged external destination cannot be presumed freed. Its full
    // reservation deliberately remains in the shared ledger (quarantine).
    for (auto& [key, entry] : entries_) drop(entry);
    if(aggregate_) { resources_->begin_eviction(aggregate_); resources_->released(aggregate_); }
}
void WeightBacking::idle() const { require(!busy_ && !transfer_, "weight_busy"); }
Footprint WeightBacking::host(Bytes bytes) const {
    auto result = resources_->snapshot().capacity;
    std::fill(result.begin(), result.end(), 0); result[0] = bytes; return result;
}
void WeightBacking::touch(Entry& entry) {
    if (clock_ == std::numeric_limits<std::uint64_t>::max()) {
        // Safe conservative tie reset, never wrap to favor the newest entry.
        for (auto& [key, item] : entries_) item.age = 0;
        clock_ = 0;
    }
    entry.age = ++clock_;
}
void WeightBacking::add(std::string key, Workload workload,
                        std::shared_ptr<const checkpoint::Shard> source, std::string tensor) {
    idle();
    require(!key.empty() && key.size() <= 256 && !tensor.empty() && tensor.size() <= 1024 && source,
            "weight_identity");
    require(workload == Workload::llm || workload == Workload::image || workload == Workload::decision, "weight_workload");
    require(entries_.size() < entry_limit_ && !entries_.contains(key), "weight_entry_limit_or_duplicate");
    source->check_unchanged();
    auto bytes = source->tensor(tensor).bytes;
    require(bytes && bytes <= SIZE_MAX && bytes <= cold_limit_ - cold_, "weight_cold_limit");
    entries_.emplace(std::move(key), Entry{workload, std::move(source), std::move(tensor), bytes, {}, 0, 0});
    cold_ += bytes;
}
void WeightBacking::drop(Entry& entry) {
    if (!entry.ram) return;
    if (!aggregate_) resources_->begin_eviction(entry.reservation);
    entry.ram.reset(); // Physical host free precedes accounting release.
    if (!aggregate_) resources_->released(entry.reservation);
    entry.reservation = 0; ram_ -= entry.bytes; count(evictions_,1);
}
bool WeightBacking::room_for_reservations(std::size_t count) {
    idle(); require(count <= Resources::max_residents, "weight_reservation_count");
    if(aggregate_) return resources_->snapshot().residents <= Resources::max_residents-count;
    while (resources_->snapshot().residents > Resources::max_residents - count) {
        auto victim=entries_.end();
        for(auto it=entries_.begin();it!=entries_.end();++it)
            if(it->second.ram&&(victim==entries_.end()||it->second.age<victim->second.age))victim=it;
        if(victim==entries_.end())return false;
        drop(victim->second);
    }
    return true;
}
void WeightBacking::evict(const std::string& key) { idle(); drop(entries_.at(key)); }
void WeightBacking::forget(const std::string& key) {
    idle(); auto& entry = entries_.at(key); drop(entry); cold_ -= entry.bytes; entries_.erase(key);
}
bool WeightBacking::retain(const std::string& key, const std::atomic_bool* cancelled) {
    idle(); Busy busy(busy_); auto& entry = entries_.at(key); cancel(cancelled);
    entry.source->check_unchanged();
    if (entry.ram) { touch(entry); return true; }
    if (entry.bytes > ram_limit_) return false; // Cold streaming, never oversubscribe RAM.
    // Evict only this store's immutable host cache. Never release external state.
    auto available = [&] {
        auto s = resources_->snapshot(); return aggregate_ ? ram_limit_ - ram_ : s.capacity[0] - s.used[0];
    };
    while (entry.bytes > ram_limit_ - ram_ || entry.bytes > available() || (!aggregate_ && resources_->snapshot().residents >= Resources::max_residents)) {
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it)
            if (it->second.ram && (victim == entries_.end() || it->second.age < victim->second.age)) victim = it;
        if (victim == entries_.end()) return false;
        drop(victim->second);
    }
    const auto reservation = aggregate_ ? aggregate_ : resources_->reserve(entry.workload, host(entry.bytes));
    try {
        auto data = std::make_unique<std::uint8_t[]>(std::size_t(entry.bytes));
        for (std::size_t at = 0; at < entry.bytes;) {
            cancel(cancelled); auto n = std::min<Bytes>(chunk_limit_, entry.bytes - at);
            entry.source->read_tensor(entry.tensor, at, {data.get() + at, std::size_t(n)}); count(source_bytes_,n); at += n;
        }
        cancel(cancelled); entry.source->check_unchanged();
        if (!aggregate_) resources_->loaded(reservation);
        entry.ram = std::move(data); entry.reservation = reservation; ram_ += entry.bytes; touch(entry);
    } catch (...) { if (!aggregate_) resources_->released(reservation); throw; }
    return true;
}
void WeightBacking::read_through(const std::string& key, Workload workload,
                                 std::shared_ptr<const checkpoint::Shard> source,
                                 const std::string& tensor, std::size_t offset,
                                 std::span<std::uint8_t> destination, const std::atomic_bool* cancelled) {
    idle(); cancel(cancelled); require(bool(source), "weight_source");
    auto found = entries_.find(key);
    if (found == entries_.end()) {
        const auto bytes = source->tensor(tensor).bytes;
        if (entries_.size() >= entry_limit_ || bytes > cold_limit_ - cold_) {
            count(misses_,1); source->read_tensor(tensor, offset, destination); count(source_bytes_,destination.size()); cancel(cancelled); return;
        }
        add(key, workload, source, tensor); found = entries_.find(key);
    }
    auto& entry = found->second;
    require(entry.source == source && entry.tensor == tensor && entry.workload == workload, "weight_source_changed");
    require(offset <= entry.bytes && destination.size() <= entry.bytes - offset, "weight_read_bounds");
    const bool hit=bool(entry.ram);
    if(hit) { count(hits_,1); count(hit_bytes_,destination.size()); }
    else count(misses_,1);
    auto state=resources_->snapshot();
    // Automatic row reads never evict another tensor to fill a miss. Alternating
    // weight/scale reads must not reload whole tensors on every row/chunk.
    const bool fits=entry.bytes<=ram_limit_-ram_ && (aggregate_ ||
        (entry.bytes<=state.capacity[0]-state.used[0] && state.residents<Resources::max_residents));
    if (hit || (fits && retain(key, cancelled))) {
        entry.source->check_unchanged(); touch(entry);
        std::copy_n(entry.ram.get() + offset, destination.size(), destination.begin());
    } else { source->read_tensor(tensor, offset, destination); count(source_bytes_,destination.size()); }
    cancel(cancelled);
}
Handle WeightBacking::begin_transfer(const std::string& key, Footprint destination) {
    idle(); auto& entry = entries_.at(key); entry.source->check_unchanged();
    const auto capacity = resources_->snapshot().capacity;
    require(destination.size() == capacity.size(), "weight_destination_shape");
    // Require enough destination bytes without summing independent GPU budgets.
    // A transfer targets one device OR host; never pool cards to fit a tensor.
    std::size_t targets = 0;
    for (auto bytes : destination) if (bytes) { require(bytes >= entry.bytes, "weight_destination_size"); ++targets; }
    require(targets == 1, "weight_destination_target");
    const auto staging = entry.ram ? 0 : std::size_t(std::min<Bytes>(entry.bytes, chunk_limit_));
    require(destination[0] <= std::numeric_limits<Bytes>::max() - staging, "weight_peak_overflow");
    destination[0] += staging;
    // Build allocating control fields before reserving.
    Transfer next{key, 0, {}, staging, false, false, false, destination};
    next.destination[0] -= staging;
    next.reservation = resources_->reserve(entry.workload, std::move(destination));
    try {
        if (staging) next.staging = std::make_unique<std::uint8_t[]>(staging);
        transfer_.emplace(std::move(next));
    } catch (...) { resources_->released(next.reservation); throw; }
    return transfer_->reservation;
}
void WeightBacking::copy(Handle id, const Sink& sink, const std::atomic_bool* cancelled) {
    require(transfer_ && transfer_->reservation == id && !transfer_->attempted && !transfer_->cleanup_started, "weight_transfer_state");
    Busy busy(busy_); transfer_->attempted = true; auto& entry = entries_.at(transfer_->key);
    cancel(cancelled); entry.source->check_unchanged();
    for (std::size_t at = 0; at < entry.bytes;) {
        cancel(cancelled); const auto n = std::size_t(std::min<Bytes>(chunk_limit_, entry.bytes - at));
        const std::uint8_t* bytes;
        if (entry.ram) bytes = entry.ram.get() + at;
        else { entry.source->read_tensor(entry.tensor, at, {transfer_->staging.get(), n}); count(source_bytes_,n); bytes = transfer_->staging.get(); }
        sink(at, {bytes, n}); at += n;
    }
    cancel(cancelled); entry.source->check_unchanged(); touch(entry);
    transfer_->complete = true;
}
Handle WeightBacking::commit(Handle id) {
    require(!busy_ && transfer_ && transfer_->reservation == id && transfer_->complete && !transfer_->cleanup_started, "weight_transfer_state");
    // Use the originally admitted footprint, subtract only this ticket's staging.
    // Reconstruct it from a saved field rather than aggregate ledger usage.
    auto destination = transfer_->destination;
    transfer_->staging.reset();
    resources_->resize_loading(id, std::move(destination));
    resources_->loaded(id);
    transfer_.reset();
    return id;
}
void WeightBacking::cleaned(Handle id, bool success) {
    require(!busy_ && transfer_ && transfer_->reservation == id, "weight_transfer_state");
    transfer_->cleanup_started = true;
    if (!success) return;
    transfer_->staging.reset();
    resources_->released(id); transfer_.reset();
}
WeightBacking::Stats WeightBacking::stats() const {
    return {ram_, cold_, transfer_ ? transfer_->bytes : 0, entries_.size(), transfer_ ? transfer_->reservation : 0, ram_limit_, hits_, misses_, hit_bytes_, source_bytes_, evictions_};
}
} // namespace kadan::serving
