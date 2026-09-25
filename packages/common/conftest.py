# pytester runs an inner pytest session: it is how the fixtures in
# common.testing are checked for skipping and failing, not just passing.
pytest_plugins = ["pytester"]
