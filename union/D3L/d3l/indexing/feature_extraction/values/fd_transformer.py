from typing import Iterable, Set

from d3l.utils.constants import (
    ALPHANUM,
    CAPALPHA,
    LOWALPHA,
    NEGDEC,
    NEGINT,
    NUMSYMB,
    POSDEC,
    POSINT,
    PUNCT,
    SYMB,
    UPPALPHA,
    WHITE,
)


class FDTransformer:
    # def __init__(self):

    @staticmethod
    def fd_tokenize(input_value: str) -> str:
        """
        Extract the format descriptor of the given input value.
        Parameters
        ----------
        input_value : str
            The input string to tokenize and extract the format descriptor from.

        Returns
        -------
        str
            A format descriptor as a string.

        """
        tokenized_value = []
        while len(input_value) > 0:
            capalphaTok = CAPALPHA.match(input_value)
            uppalphaTok = UPPALPHA.match(input_value)
            lowalphaTok = LOWALPHA.match(input_value)

            posdecTok = POSDEC.match(input_value)
            negdecTok = NEGDEC.match(input_value)
            posintTok = POSINT.match(input_value)
            negintTok = NEGINT.match(input_value)

            punctTok = PUNCT.match(input_value)
            symbTok = SYMB.match(input_value)
            whiteTok = WHITE.match(input_value)

            alphanumTok = ALPHANUM.match(input_value)
            numsymbTok = NUMSYMB.match(input_value)

            if alphanumTok:
                tok = alphanumTok.group()
                tok_type = "a"
            elif posdecTok:
                tok = posdecTok.group()
                tok_type = "d"
            elif negdecTok:
                tok = negdecTok.group()
                tok_type = "e"
            elif posintTok:
                tok = posintTok.group()
                tok_type = "i"
            elif negintTok:
                tok = negintTok.group()
                tok_type = "j"
            elif numsymbTok:
                tok = numsymbTok.group()
                tok_type = "q"
            elif capalphaTok:
                tok = capalphaTok.group()
                tok_type = "c"
            elif uppalphaTok:
                tok = uppalphaTok.group()
                tok_type = "u"
            elif lowalphaTok:
                tok = lowalphaTok.group()
                tok_type = "l"
            elif punctTok:
                tok = punctTok.group()
                tok_type = "p"
            elif symbTok:
                tok = symbTok.group()
                tok_type = "s"
            elif whiteTok:
                tok = whiteTok.group()
                tok_type = "w"
            else:
                break

            tokenized_value.append(tok_type)
            input_value = input_value[len(tok) :]

        pattern = "".join(tokenized_value)

        # pattern = re.sub(r"w?([cula])w?", r"\1", pattern)
        # pattern = re.sub(r"([cula])\1+", r"\1", pattern)

        return pattern

    @staticmethod
    def transform(input_values: Iterable[str]) -> Set[str]:
        """
        Generate a collection of format descriptors denoting all the formats available in the input.
        Parameters
        ----------
        input_values : Iterable[str]
            A collection of values.

        Returns
        -------
        Set[str]
            A set of format descriptors.
        """

        if len(input_values) <= 0:
            return set()
        fds = {
            FDTransformer.fd_tokenize(str(value).replace("\n", " ").strip())
            for value in input_values
            if str(value) != ""
        }

        return fds
