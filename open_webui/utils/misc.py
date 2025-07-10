"""Mock implementation of open_webui.utils.misc for testing"""


def pop_system_message(messages):
    """
    Mock implementation of pop_system_message.
    Removes and returns the first system message from the messages list.
    """
    system_message = None
    filtered_messages = []

    for message in messages:
        if message.get("role") == "system" and system_message is None:
            system_message = message
        else:
            filtered_messages.append(message)

    return system_message, filtered_messages
