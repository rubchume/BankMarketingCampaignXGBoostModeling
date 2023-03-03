from inspect import signature


def dataframe_input(function):
    def dataframe_to_dictionary_of_series(dataframe):
        return dict(dataframe.items())

    def wrapper(dataframe):
        kwargs = dataframe_to_dictionary_of_series(dataframe)
        return accept_kwargs(function)(**kwargs)

    return wrapper


def accept_kwargs(function):
    def wrapper(**named_arguments):
        accepted_parameters = {
            parameter: named_arguments[parameter]
            for parameter in signature(function).parameters.keys()
        }

        return function(**accepted_parameters)

    return wrapper
