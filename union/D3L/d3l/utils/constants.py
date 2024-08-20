import re
import unicodedata

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

SYMBPATT = r"\@" + re.escape(
    u"".join(chr(i) for i in range(0xFFFF) if unicodedata.category(chr(i)) == "Sc")
)
PUNCTPATT = r"\!\"\#\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\[\\\]\^\_\`\{\|\}\~"

LOWALPHA = re.compile(r"[a-z]([a-z\-])*")
UPPALPHA = re.compile(r"[A-Z]([A-Z\-\.])*")
CAPALPHA = re.compile(r"[A-Z][a-z]([a-z\-])*")

POSDEC = re.compile(r"\+?[0-9]+(,[0-9]+)*\.[0-9]+")
NEGDEC = re.compile(r"\-[0-9]+(,[0-9]+)*\.[0-9]+")
POSINT = re.compile(r"\+?[0-9]+(,[0-9]+)*")
NEGINT = re.compile(r"\-[0-9]+(,[0-9]+)*")

PUNCT = re.compile(r"[" + PUNCTPATT + r"]+")
SYMB = re.compile(r"[" + SYMBPATT + r"]+", re.UNICODE)
WHITE = re.compile(r"\s+")

ALPHANUM = re.compile(r"(?:[0-9]+[a-zA-Z]|[a-zA-Z]+[0-9])[a-zA-Z0-9]*")
NUMSYMB = re.compile(
    r"(?=.*[0-9,\.])(?=.*[" + SYMBPATT + r"]+)([0-9" + SYMBPATT + r"]+)", re.UNICODE
)

FASTTEXTURL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/"
GLOVEURL = "http://nlp.stanford.edu/data/"