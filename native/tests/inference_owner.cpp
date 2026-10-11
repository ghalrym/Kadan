#include "kadan/inference_owner.hpp"
#include <cassert>
#include <iostream>
#include <tuple>
using namespace kadan;
using namespace kadan::serving;
struct FakeState {
  std::vector<std::string> events;
  std::vector<Footprint> loads;
  bool wait_once = false, fail_close = false;
  unsigned preparations = 0, close_calls = 0;
  std::shared_ptr<Resources> ledger;
};
struct FakeEngine : InferenceEngine {
  FakeState &state;
  std::shared_ptr<Resources> ledger;
  Handle handle = 0;
  std::string name;
  FakeEngine(FakeState &s, std::shared_ptr<Resources> r)
      : state(s), ledger(std::move(r)) {
    if (state.ledger)
      assert(state.ledger == ledger);
    else
      state.ledger = ledger;
  }
  void load(const InferenceRequest &request, const Footprint &free,
            const std::atomic_bool &) override {
    name = request.model;
    state.events.push_back("load:" + name);
    state.loads.push_back(free);
    handle = ledger->reserve(request.workload, {10, 0, 0});
    if (state.wait_once) {
      state.wait_once = false;
      throw AdmissionWait("physical_pressure");
    }
  }
  void prepare(const InferenceRequest &, const Footprint &,
               const std::atomic_bool &) override {
    ++state.preparations;
  }
  std::string execute(const InferenceRequest &,
                      const std::atomic_bool &) override {
    state.events.push_back("execute:" + name);
    return name;
  }
  void close() override {
    ++state.close_calls;
    if (state.fail_close)
      throw std::runtime_error("cleanup_failed");
    if (handle) {
      ledger->released(handle);
      handle = 0;
      state.events.push_back("close:" + name);
    }
  }
};
InferenceRequest request(std::string id, Workload workload = Workload::image) {
  return {id, id, Operation::generate, "{}", workload};
}
void fifo_shares_ledger() {
  FakeState state;
  std::vector<std::tuple<std::string, JobState, std::string>> replies;
  Footprint free{100, 80, 70};
  InferenceOwner owner(
      {100, 100, 100},
      [&](auto, std::shared_ptr<Resources> r) {
        return std::make_unique<FakeEngine>(state, r);
      },
      [&] { return free; },
      [&](auto id, auto kind, auto value) {
        replies.emplace_back(id, kind, value);
      });
  const Workload workloads[]{Workload::llm,    Workload::decision,
                             Workload::image,  Workload::video,
                             Workload::speech, Workload::tts};
  for (std::size_t i = 0; i < 6; ++i)
    owner.submit(request(std::to_string(i), workloads[i]));
  for (std::size_t i = 0; i < 6; ++i) {
    assert(owner.advance());
    assert(std::get<0>(replies.back()) == std::to_string(i));
  }
  assert(owner.pending() == 0);
  owner.shutdown();
  assert(owner.snapshot().residents == 0);
  // Reclaim overestimation cannot carry across handoff: the probe occurs after
  // the actual previous engine cleanup, and only its measured free bytes enter
  // load.
  assert(state.loads.size() == 6);
  for (auto p : state.loads)
    assert(p == free);
}

void admission_retries_head() {
  std::vector<std::tuple<std::string, JobState, std::string>> replies;
  FakeState retry;
  retry.wait_once = true;
  std::size_t probes = 0;
  replies.clear();
  InferenceOwner pending(
      {100, 100, 100},
      [&](auto, std::shared_ptr<Resources> r) {
        return std::make_unique<FakeEngine>(retry, r);
      },
      [&] {
        return ++probes == 1 ? Footprint{100, 80, 70} : Footprint{100, 40, 30};
      },
      [&](auto id, auto kind, auto value) {
        replies.emplace_back(id, kind, value);
      });
  pending.submit(request("head"));
  pending.submit(request("tail"));
  assert(!pending.advance());
  assert(pending.pending() == 2);
  assert(retry.events == std::vector<std::string>({"load:head", "close:head"}));
  assert(pending.advance());
  assert(retry.loads.back() == Footprint({100, 40, 30}));
  assert(std::get<0>(replies.back()) == "head" &&
         std::get<1>(replies.back()) == JobState::succeeded);
  assert(pending.cancel("tail"));
  assert(pending.advance());
  assert(std::get<1>(replies.back()) == JobState::cancelled);
  pending.shutdown();
}

void cleanup_failure_stops_owner() {
  FakeState broken;
  InferenceOwner quarantine(
      {100, 100, 100},
      [&](auto, std::shared_ptr<Resources> r) {
        return std::make_unique<FakeEngine>(broken, r);
      },
      [] { return Footprint{100, 80, 70}; }, [](auto, auto, auto) {});
  quarantine.submit(request("first"));
  assert(quarantine.advance());
  broken.fail_close = true;
  quarantine.submit(request("second"));
  bool failed = false;
  try {
    quarantine.advance();
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed);
  assert(quarantine.pending() == 1 && quarantine.snapshot().residents == 1);
  failed = false;
  try {
    quarantine.advance();
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed);
  assert(broken.loads.size() == 1 && broken.close_calls == 1);
}

void reused_model_prepares_each_request() {
  FakeState same;
  InferenceOwner reusable(
      {100, 100, 100},
      [&](auto, std::shared_ptr<Resources> r) {
        return std::make_unique<FakeEngine>(same, r);
      },
      [] { return Footprint{100, 80, 70}; }, [](auto, auto, auto) {});
  auto first = request("one");
  first.model = "same";
  auto second = request("two");
  second.model = "same";
  reusable.submit(first);
  reusable.submit(second);
  assert(reusable.advance());
  assert(reusable.advance());
  assert(same.loads.size() == 1 && same.preparations == 2);
  reusable.shutdown();
}

void terminal_delivery_never_replays() {
  bool failed = false;
  FakeState delivery;
  InferenceOwner transport(
      {100, 100, 100},
      [&](auto, std::shared_ptr<Resources> r) {
        return std::make_unique<FakeEngine>(delivery, r);
      },
      [] { return Footprint{100, 80, 70}; },
      [](auto, auto state, auto) {
        if (state == JobState::succeeded)
          throw std::runtime_error("reader_gone");
      });
  transport.submit(request("once"));
  failed = false;
  try {
    transport.advance();
  } catch (const PublicationFailure &) {
    failed = true;
  }
  assert(failed && transport.pending() == 0);
  failed = false;
  try {
    transport.advance();
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed);
  assert(std::count(delivery.events.begin(), delivery.events.end(),
                    "execute:once") == 1);
}

void preparation_keeps_fifo_head_until_acknowledged() {
  FakeState state;
  std::vector<JobState> events;
  InferenceOwner owner(
      {100, 100, 100},
      [&](auto, auto resources) {
        return std::make_unique<FakeEngine>(state, resources);
      },
      [] { return Footprint{100, 80, 70}; },
      [&](auto, auto state, auto) { events.push_back(state); });
  auto head = request("head");
  head.prepared = false;
  auto tail = request("tail");
  tail.prepared = false;
  owner.submit(head);
  owner.submit(tail);
  assert(!owner.advance());
  assert(events.back() == JobState::preparing);
  owner.prepared("tail", "{}");
  assert(!owner.advance());
  assert(state.loads.empty());
  owner.cancel("head");
  assert(!owner.advance());
  assert(owner.pending() == 2);
  owner.prepared("head", "{}", "cancelled");
  assert(owner.advance());
  assert(events.back() == JobState::cancelled);
  assert(owner.advance());
  assert(state.loads.size() == 1);
  owner.shutdown();
}

void changed_configuration_reloads_and_gpu_pressure_reconciles() {
  FakeState state;
  Footprint free{100, 80, 70};
  InferenceOwner owner(
      {100, 100, 100},
      [&](auto, auto resources) {
        return std::make_unique<FakeEngine>(state, resources);
      },
      [&] { return free; }, [](auto, auto, auto) {});
  auto first = request("first", Workload::decision);
  first.model = "same";
  first.configuration = "8k";
  owner.submit(first);
  assert(owner.advance());
  auto second = first;
  second.id = "second";
  free[1] = 79;
  owner.submit(second);
  assert(owner.advance());
  assert(state.loads.size() == 1);
  assert(owner.snapshot().capacity[1] == 79);
  auto third = first;
  third.id = "third";
  third.configuration = "64k";
  owner.submit(third);
  assert(owner.advance());
  assert(state.loads.size() == 2);
  owner.shutdown();
}

void refreshed_reservations_never_create_physical_capacity() {
  Resources resources({100});
  auto resident = resources.reserve(Workload::llm, {40});
  // Only ten physical bytes back a conservative forty-byte reservation.
  resources.refresh_available({90});
  assert(resources.snapshot().capacity[0] == 100);
  resources.released(resident);
  bool rejected = false;
  try {
    resources.reserve(Workload::llm, {130});
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  assert(rejected);
}

void resident_gpu_pressure_replans_before_execution() {
  struct PlannedEngine : FakeEngine {
    Bytes planned = 0;
    using FakeEngine::FakeEngine;
    void load(const InferenceRequest &request, const Footprint &available,
              const std::atomic_bool &cancel) override {
      FakeEngine::load(request, available, cancel);
      planned = available[1];
    }
    void prepare(const InferenceRequest &, const Footprint &available,
                 const std::atomic_bool &) override {
      if (available[1] < planned)
        throw AdmissionWait("request_state_pressure");
    }
  };
  FakeState state;
  Footprint free{100, 80, 70};
  std::vector<JobState> events;
  InferenceOwner owner(
      {100, 100, 100},
      [&](auto, auto ledger) {
        return std::make_unique<PlannedEngine>(state, ledger);
      },
      [&] { return free; },
      [&](auto, auto event, auto) { events.push_back(event); });
  auto head = request("first");
  head.model = "same";
  owner.submit(head);
  assert(owner.advance());
  free[1] = 40;
  head.id = "second";
  owner.submit(head);
  assert(!owner.advance());
  assert(events.back() == JobState::waiting_for_resources);
  assert(owner.pending() == 1);
  assert(state.close_calls == 1);
  assert(owner.advance());
  assert(events.back() == JobState::succeeded);
  assert(state.loads.size() == 2 && state.loads.back()[1] == 40);
  owner.shutdown();
}

int main() {
  resident_gpu_pressure_replans_before_execution();
  refreshed_reservations_never_create_physical_capacity();
  preparation_keeps_fifo_head_until_acknowledged();
  changed_configuration_reloads_and_gpu_pressure_reconciles();
  fifo_shares_ledger();
  admission_retries_head();
  cleanup_failure_stops_owner();
  reused_model_prepares_each_request();
  terminal_delivery_never_replays();
  std::cout << "Native owner fake-engine units passed\n";
}
