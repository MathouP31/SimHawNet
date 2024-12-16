from .loss import NLLLoss


def create_model(name, *args, **kwargs):
    
    if name == "poisson":
        from .poisson import Poisson
        return Poisson(
            *args,
            **kwargs,
        )

    if name == "markov":
        from .pwc_markov import MarkovPieceWiseConst
        return MarkovPieceWiseConst(
            *args,
            **kwargs,
        )

    if name == "markov_exp":
        from .pwc_markov import MarkovExp
        return MarkovExp(
            *args,
            **kwargs,
        )
