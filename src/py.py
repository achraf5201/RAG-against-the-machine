import fire

def hello(name="World"):
    """Says hello to the specified name."""
    return f"Hello {name}!"

if __name__ == '__main__':
    fire.Fire()
