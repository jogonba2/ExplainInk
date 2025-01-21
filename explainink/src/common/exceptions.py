class ExplainInkError(Exception):
    """
    Base class for ExplainInk exceptions.
    """

    ...


class InvalidModelForExplainer(ExplainInkError):
    """
    Raised when a model can't be used with an explainer.
    """

    def __init__(self, model_name: str, explainer_name: str):
        self.model_name = model_name
        self.explainer_name = explainer_name
        msg = (
            f"'{self.model_name}' can't be used for the"
            f" explainer {self.explainer_name}."
        )
        super().__init__(msg)
