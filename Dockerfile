FROM pytorch/pytorch

# Install OpenFed
COPY . /OpenFed
WORKDIR /OpenFed
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
