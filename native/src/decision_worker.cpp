#include "kadan/decision_executor.hpp"
#include "kadan/generation_queue.hpp"
#include <cblas.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <string>
#include <unistd.h>

namespace {
std::atomic_bool cancelled{false};
void stop(int) { cancelled.store(true); }
using Queue = kadan::serving::GenerationQueue;

// Synchronous executor ownership lasts through completion and cleanup. The
// transport supplies request order; the shared queue owns resident admission.
class Driver {
  public:
    Driver(std::shared_ptr<kadan::Resources> resources, std::string root)
        : queue_(resources), executor_(resources), root_(std::move(root)) {}
    ~Driver() { shutdown(); }
    std::string execute(const std::string &request) {
        auto id = queue_.submit(
            {root_, kadan::Workload::decision, {kadan::decision::Executor::envelope_bytes}});
        std::exception_ptr error;
        std::string result;
        while (queue_.pending()) {
            auto action = queue_.poll();
            if (action.kind == Queue::Kind::load) {
                try {
                    executor_.load(root_, cancelled, action.reservation);
                } catch (...) {
                    error = std::current_exception();
                }
                queue_.loaded(id, !error);
            } else if (action.kind == Queue::Kind::execute) {
                try {
                    result = executor_.execute(request, cancelled);
                } catch (...) {
                    error = std::current_exception();
                }
                if (cancelled.load())
                    queue_.cancel(id);
                queue_.completed(id, !error);
            } else if (action.kind == Queue::Kind::cleanup) {
                executor_.unload(); // Free physical allocations before acknowledgement.
                queue_.cleaned(action.reservation, true);
            } else {
                throw std::runtime_error("decision_queue_stalled");
            }
        }
        if (error)
            std::rethrow_exception(error);
        if (cancelled.load())
            throw std::runtime_error("decision_cancelled");
        return result;
    }
    void shutdown() {
        queue_.stop();
        auto action = queue_.poll();
        if (action.kind == Queue::Kind::cleanup) {
            executor_.unload();
            queue_.cleaned(action.reservation, true);
        }
    }

  private:
    Queue queue_;
    kadan::decision::Executor executor_;
    std::string root_;
};

// Polling stdin lets SIGTERM cancel an idle worker as well as model execution.
// A line is one request; no later input can overtake its execution/publication.
bool next_byte(char &value) {
    while (!cancelled.load()) {
        pollfd input{STDIN_FILENO, POLLIN, 0};
        int ready = poll(&input, 1, 100);
        if (ready < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error("decision_stdin_poll");
        }
        if (!ready)
            continue;
        ssize_t count = read(STDIN_FILENO, &value, 1);
        if (count == 1)
            return true;
        if (count == 0)
            return false;
        if (errno != EINTR)
            throw std::runtime_error("decision_stdin_read");
    }
    return false;
}
// A stalled reader cannot prevent cancellation or resident cleanup. A partial
// frame is never retried as another response; publication failure ends the worker.
void publish_frame(const std::string &result) {
    std::string frame = result + '\n';
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    std::size_t offset = 0;
    while (offset < frame.size()) {
        if (cancelled.load())
            return;
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("decision_stdout_timeout");
        pollfd output{STDOUT_FILENO, POLLOUT, 0};
        int ready = poll(&output, 1, 100);
        if (ready < 0 && errno == EINTR)
            continue;
        if (ready < 0 || (output.revents & (POLLERR | POLLHUP | POLLNVAL)))
            throw std::runtime_error("decision_stdout_poll");
        if (!ready)
            continue;
        auto count = write(STDOUT_FILENO, frame.data() + offset, frame.size() - offset);
        if (count > 0)
            offset += std::size_t(count);
        else if (count < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
            throw std::runtime_error("decision_stdout_write");
    }
}
} // namespace

int main(int argc, char **argv) {
    static_assert(std::atomic_bool::is_always_lock_free);
    // Error reporting must not flush a failed stdout stream.
    std::cerr.tie(nullptr);
    try {
        if (argc != 2)
            throw std::runtime_error("usage: kadan-decision-worker CHECKPOINT_ROOT (one request "
                                     "JSON per line on stdin)");
        openblas_set_num_threads(1);
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        std::signal(SIGPIPE, SIG_IGN);
        int flags = fcntl(STDOUT_FILENO, F_GETFL);
        if (flags < 0 || fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK) < 0)
            throw std::runtime_error("decision_stdout_nonblocking");
        constexpr kadan::Bytes io_bytes = 1024 * 1024;
        auto resources = std::make_shared<kadan::Resources>(
            kadan::Footprint{kadan::decision::Executor::envelope_bytes + io_bytes});
        auto io = resources->reserve(kadan::Workload::decision, {io_bytes});
        std::exception_ptr failure;
        try {
            Driver driver(resources, std::filesystem::canonical(argv[1]).string());
            auto publish = [&](const std::string &line) {
                std::string result;
                try {
                    result = driver.execute(line);
                } catch (const std::exception &e) {
                    result = nlohmann::json{{"error", e.what()}}.dump();
                }
                if (!cancelled.load())
                    publish_frame(result);
            };
            std::string line;
            char value;
            while (next_byte(value)) {
                if (value != '\n') {
                    if (line.size() >= 65536)
                        throw std::runtime_error("decision_request_size");
                    line += value;
                } else if (!line.empty()) {
                    publish(line);
                    line.clear();
                }
            }
            if (!line.empty() && !cancelled.load())
                publish(line);
            driver.shutdown();
        } catch (...) {
            failure = std::current_exception();
        }
        resources->released(io);
        std::cerr << "resident_bytes=" << resources->snapshot().used[0] << '\n';
        if (failure)
            std::rethrow_exception(failure);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
