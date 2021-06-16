from openfed import Backend
import time
backend = Backend()
print("#" * 40)

time.sleep(1.0)

backend.finish()
print("Test success of backend.")
