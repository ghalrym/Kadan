"""Validate API imports without starting the application or loading a model."""
from api.services.native_worker import NativeWorker
from api.services.inference import InferenceService


def main():
    assert NativeWorker and InferenceService
    print('Single native worker transport imports successfully; no model execution.')


if __name__ == '__main__':
    main()
