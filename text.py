import hyperparams as hp


char_to_id = {char: i for i, char in enumerate(hp.chars)}
id_to_char = {i: char for i, char in enumerate(hp.chars)}


def text_to_sequence(text, eos=hp.eos):
    """
    Args:
        text:
        eos: End of string token.

    Returns:

    """
    text += eos
    return [char_to_id[char] for char in text]


def sequence_to_text(sequence):
    return "".join(id_to_char[i] for i in sequence)