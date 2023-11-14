def DSP(predictions, groups, positive_prediction=1, previledged_key=1):
    """
    Calculate the Disparate Impact of a predictive model.

    Args:
        predictions (List[any]): A list of predictions for each observation.
        groups (List[any]): A list of group identifiers for each observation.
        positive_prediction (any, optional): The value representing the positive prediction. Defaults to 1.
        previledged_key (any, optional): The value representing the privileged group. Defaults to 1.

    Returns:
        float: The Disparate Impact score.
    """
    if positive_prediction != 1:
        predictions = [1 if x == positive_prediction else 0 for x in predictions]

    if previledged_key != 1:
        groups = [1 if x == previledged_key else 0 for x in groups]

    assert len(predictions) == len(groups)
    assert set(groups).issubset({0, 1})
    assert set(predictions).issubset({0, 1})

    total_previledged = len(
        [1 for idx, fy in enumerate(predictions) if groups[idx] == 1]
    )
    total_unpreviledged = len(
        [1 for idx, fy in enumerate(predictions) if groups[idx] == 0]
    )

    if total_previledged == 0 or total_unpreviledged == 0:
        return -1

    p_previledged = (
        float(
            len(
                [
                    1
                    for idx, fy in enumerate(predictions)
                    if fy == 1 and groups[idx] == 1
                ]
            )
        )
        / total_previledged
    )

    p_unpreviledged = (
        float(
            len(
                [
                    1
                    for idx, fy in enumerate(predictions)
                    if fy == 1 and groups[idx] == 0
                ]
            )
        )
        / total_unpreviledged
    )

    return abs(p_previledged - p_unpreviledged)


def DEO(model, X, Y, groups):
    # model: the trained model
    # X: our data of n examples with d features
    # Y: binary labels of our n examples (1 = positive)
    # groups: a list of n values binary values defining two different subgroups of the populations

    fY = model.predict(X)
    eo = [0, 0]
    eo[0] = float(
        len(
            [
                1
                for idx, fy in enumerate(fY)
                if fy == 1 and groups[idx] == 0 and Y[idx] == 1
            ]
        )
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 0 and Y[idx] == 1])
    eo[1] = float(
        len(
            [
                1
                for idx, fy in enumerate(fY)
                if fy == 1 and groups[idx] == 1 and Y[idx] == 1
            ]
        )
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 1 and Y[idx] == 1])
    return abs(eo[0] - eo[1])
