# Present so pytest adds this directory to sys.path and "app" resolves to
# this service. Every service ships a package under that name, which is
# why tests are run from inside the service directory rather than from
# the repository root.
