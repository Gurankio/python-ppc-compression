class Tokenizer:

    @staticmethod
    def tokenize(content: str, width: int = 1, len_limit: int = 10):
        # Get a list of lines from the content of the file
        tokens = content.splitlines()

        if width > 1:
            # Tokes are consecutive lines grouped together
            tokens = [tokens[i:i + width][0] for i in range(max(len(tokens) - width + 1, 1))]

        # Remove lines with less than `len_limit` chars and delete leading and trailing tabs and whitespaces
        tokens = [x.strip() for x in tokens if len(x) > len_limit]
        return tokens
