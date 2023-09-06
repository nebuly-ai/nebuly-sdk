from queue import Queue

from nebuly.publisher import Publisher


def test_worker() -> None:
    messages = [
        "test message 1",
        "test message 2",
        "test message 3",
    ]
    results = []
    queue = Queue()
    publisher = Publisher(queue, results.append)

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
    results = []
    queue = Queue()
    for message in messages:
        queue.put(message)

    publisher = Publisher(queue, results.append)
    publisher.stop()

    assert results == messages
