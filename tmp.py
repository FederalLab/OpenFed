from openfed.common.vars import DEBUG
import openfed

print(DEBUG)
# openfed.debug()
DEBUG = True
print(id(openfed.DEBUG), openfed.DEBUG)
print(id(DEBUG), DEBUG)