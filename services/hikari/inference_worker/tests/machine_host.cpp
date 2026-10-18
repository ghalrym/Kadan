// Command routing with fake engines: no model acquisition or inference.
#include "kadan/machine_host.hpp"

#include <cassert>
#include <iostream>

using namespace kadan;
using namespace kadan::serving;

namespace {
struct FakeEngine : InferenceEngine {
    std::shared_ptr<Resources> ledger;
    Handle handle = 0;

    explicit FakeEngine(std::shared_ptr<Resources> r) : ledger(std::move(r)) {
    }

    void load(
        const InferenceRequest&, const Footprint& free, const std::atomic_bool&
    ) override {
        Footprint reservation(free.size());
        reservation[0] = 1;
        handle = ledger->reserve(Workload::llm, reservation);
    }

    void prepare(
        const InferenceRequest&, const Footprint&, const std::atomic_bool&
    ) override {
    }

    Json execute(const InferenceRequest& request, const std::atomic_bool&) override {
        return {{"model", request.model}};
    }

    void close() override {
        if (handle)
            ledger->released(handle);
        handle = 0;
    }
};

struct Harness {
    std::vector<Json> sent;
    std::vector<std::string> prepared, cleaned;
    std::unique_ptr<MachineHost> host;

    explicit Harness(std::vector<std::string> names = {}) {
        MachineHost::Hooks hooks;
        hooks.send = [this](const Json& message) {
            sent.push_back(message);
        };
        hooks.factory = [](Workload, std::shared_ptr<Resources> r, WorkerChannel&) {
            return std::make_unique<FakeEngine>(std::move(r));
        };
        hooks.prepare = [this](
                            InferenceRequest& request,
                            const std::atomic_bool&,
                            WorkerChannel& channel
                        ) {
            prepared.push_back(channel.name + ":" + request.id);
            request.model = "fake";
        };
        hooks.probe = [] {
            return Footprint{1000, 1000};
        };
        hooks.requirement = [](Workload,
                               const std::optional<std::string>& model,
                               const std::optional<Json>& context) {
            return Json{
                {"model", model.value_or("default")},
                {"context", context ? *context : Json("saved")},
                {"minimum", {1, 2}},
                {"preferred", {3, 4}}
            };
        };
        hooks.machine = [] {
            return Json{{"name", "box"}};
        };
        hooks.models = [] {
            return Json{{"models", Json::array()}};
        };
        hooks.cleanup = [this](const std::string& id) {
            cleaned.push_back(id);
        };
        host = std::make_unique<MachineHost>(worker_specs(names), 1, hooks);
    }

    Json last(const std::string& id) const {
        for (auto it = sent.rbegin(); it != sent.rend(); ++it)
            if (it->value("id", "") == id)
                return *it;
        return nullptr;
    }

    void command(Json message) {
        host->handle(message.dump());
    }
};

std::string id(char c) {
    return std::string(32, c);
}

Json chat(const std::string& request_id) {
    return {
        {"id", request_id},
        {"command", "submit"},
        {"operation", "completion"},
        {"input",
         {{"feature", "llm"}, {"messages", {{{"role", "user"}, {"text", "hi"}}}}}}
    };
}

void inference_waits_for_a_budget_on_its_own_worker() {
    Harness harness;
    harness.command(chat(id('a')));
    assert(harness.last(id('a'))["state"] == "queued");
    assert(harness.last(id('a'))["worker"] == "llm");
    assert(!harness.host->advance_all());
    harness.command(
        {{"id", id('b')},
         {"command", "load"},
         {"worker", "llm"},
         {"budget", {500, 600}}}
    );
    const auto loaded = harness.last(id('b'));
    assert(loaded["state"] == "control" && loaded["ok"] == true);
    while (harness.host->advance_all()) {
    }
    const auto done = harness.last(id('a'));
    assert(done["state"] == "succeeded" && done["worker"] == "llm");
    assert(done["result"]["model"] == "fake");
    assert(harness.prepared == std::vector<std::string>({"llm:" + id('a')}));
    assert(harness.cleaned == std::vector<std::string>({id('a')}));
    harness.host->stop();
}

void downloads_and_settings_run_on_the_store_worker() {
    Harness harness;
    harness.command(
        {{"id", id('c')},
         {"command", "submit"},
         {"operation", "download"},
         {"input", {{"feature", "models"}, {"model_id", "laya"}}}}
    );
    harness.command(
        {{"id", id('d')},
         {"command", "submit"},
         {"operation", "configure"},
         {"input", {{"feature", "tier_selection"}, {"tier", "go"}}}}
    );
    while (harness.host->advance_all()) {
    }
    assert(harness.last(id('c'))["state"] == "succeeded");
    assert(harness.last(id('c'))["worker"] == "store");
    assert(harness.last(id('d'))["worker"] == "store");
    harness.host->stop();
}

void status_offload_and_requirement_replies() {
    Harness harness({"llm", "image"});
    harness.command({{"id", id('e')}, {"command", "status"}});
    const auto status = harness.last(id('e'));
    assert(status["state"] == "status" && status["machine"]["name"] == "box");
    assert(status["workers"].size() == 3 && status["workers"][1]["worker"] == "llm");
    assert(status["workers"][1]["loaded"] == false);
    assert(status["workers"][1]["releasing"] == false);
    harness.command(
        {{"id", id('f')}, {"command", "load"}, {"worker", "image"}, {"budget", {1, 1}}}
    );
    harness.command({{"id", id('0')}, {"command", "is_busy"}, {"worker", "image"}});
    assert(harness.last(id('0'))["busy"] == false);
    harness.command({{"id", id('1')}, {"command", "offload"}, {"worker", "image"}});
    assert(harness.last(id('1'))["ok"] == true);
    harness.host->advance_all();
    const auto& event = harness.sent.back();
    assert(event["state"] == "worker" && event["event"] == "offloaded");
    assert(event["worker"] == "image" && event["loaded"] == false);
    harness.command(
        {{"id", id('2')},
         {"command", "requirement"},
         {"worker", "llm"},
         {"model", "qwen"}}
    );
    const auto requirement = harness.last(id('2'));
    assert(requirement["state"] == "requirement" && requirement["worker"] == "llm");
    assert(requirement["model"] == "qwen" && requirement["preferred"][1] == 4);
    assert(requirement["context"] == "saved");
    harness.command(
        {{"id", id('8')},
         {"command", "requirement"},
         {"worker", "llm"},
         {"context", 8192}}
    );
    assert(harness.last(id('8'))["context"] == 8192);
    harness.command(
        {{"id", id('9')},
         {"command", "requirement"},
         {"worker", "llm"},
         {"context", nullptr}}
    );
    assert(harness.last(id('9'))["context"].is_null());
    harness.host->stop();
}

void invalid_commands_are_rejected_not_thrown() {
    Harness harness({"llm"});
    harness.command(
        {{"id", id('3')}, {"command", "load"}, {"worker", "llm"}, {"budget", {1}}}
    );
    assert(harness.last(id('3'))["state"] == "rejected");
    assert(harness.last(id('3'))["status_code"] == 422);
    harness.command(
        {{"id", id('4')}, {"command", "load"}, {"worker", "video"}, {"budget", {1, 1}}}
    );
    assert(harness.last(id('4'))["status_code"] == 404);
    harness.command({{"id", id('5')}, {"command", "offload"}, {"worker", "store"}});
    assert(harness.last(id('5'))["state"] == "rejected");
    harness.command(
        {{"id", id('6')},
         {"command", "submit"},
         {"operation", "generate"},
         {"input", {{"feature", "image"}, {"prompt", "fox"}, {"count", 1}}}}
    );
    assert(harness.last(id('6'))["state"] == "rejected");
    harness.host->handle("{\"id\": \"short\", \"command\": \"status\"}");
    assert(harness.sent.back()["state"] == "rejected");
    harness.host->handle("not json");
    assert(harness.sent.back()["state"] == "rejected");
    harness.command({{"id", id('7')}, {"command", "stop"}});
    assert(harness.host->stopping());
    harness.host->stop();
}

void worker_names_are_validated() {
    const auto specs = worker_specs({"image", "llm"});
    assert(specs.size() == 3 && specs[0].name == "store" && !specs[0].workload);
    assert(specs[1].name == "llm" && specs[2].name == "image");
    assert(worker_specs({}).size() == 8);
    bool rejected = false;
    try {
        worker_specs({"gpu"});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}
}  // namespace

int main() {
    inference_waits_for_a_budget_on_its_own_worker();
    downloads_and_settings_run_on_the_store_worker();
    status_offload_and_requirement_replies();
    invalid_commands_are_rejected_not_thrown();
    worker_names_are_validated();
    std::cout << "Machine host units passed\n";
}
