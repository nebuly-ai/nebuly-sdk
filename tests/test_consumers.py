from queue import Queue

from nebuly.consumers import ConsumerWorker


def test_worker() -> None:
    messages = [
        "test message 1",
        "test message 2",
        "test message 3",
    ]
    results: list[str] = []
    queue: Queue[str] = Queue()
    publisher = ConsumerWorker(queue, results.append)  # type: ignore

    for message in messages:
        queue.put(message)

    queue.join()

    assert results == messages

    publisher.stop()


def test_worker_publish_messages_before_exit() -> None:
    messages = [
        "test message 1",
        "test message 2",
        "test message 3",
    ]
    results: list[str] = []
    queue: Queue[str] = Queue()
    for message in messages:
        queue.put(message)

    publisher = ConsumerWorker(queue, results.append)  # type: ignore
    publisher.stop()

    assert results == messages
