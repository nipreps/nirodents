FROM nipreps/miniconda:py38_1.4.2

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    autoconf \
                    build-essential \
                    bzip2 \
                    ca-certificates \
                    curl \
                    git \
                    libtool \
                    lsb-release \
                    pkg-config \
                    unzip \
                    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Installing ANTs 2.3.0 (NeuroDocker build)
ENV ANTSPATH=/usr/lib/ants
RUN mkdir -p $ANTSPATH && \
    curl -sSL "https://dl.dropbox.com/s/hrm530kcqe3zo68/ants-Linux-centos6_x86_64-v2.3.2.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1
ENV PATH=$ANTSPATH/bin:$PATH

# WORKDIR /opt/pcnn3d
# RUN curl -sSL "https://f495cb51-a-62cb3a1a-s-sites.googlegroups.com/site/chuanglab/software/3d-pcnn/PCNN3D%20binary.zip" -o "pcnn3d.zip" && \
#     unzip pcnn3d.zip && \
#     rm pcnn3d.zip && \
#     chmod a+rx PCNNBrainExtract
# ENV PATH="/opt/pcnn3d:$PATH"

# Uncomment these lines for RATS (requires the software bundle)
# WORKDIR /opt/RATS
# COPY docker/files/rats.tar.gz /tmp/
# RUN tar xzf /tmp/rats.tar.gz
# ENV PATH="/opt/RATS/distribution:$PATH"

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users nirodents
WORKDIR /home/nirodents
ENV HOME="/home/nirodents"

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    TEMPLATEFLOW_AUTOUPDATE=0

# Installing dev requirements (packages that are not in pypi)
WORKDIR /src/
COPY . nirodents/
WORKDIR /src/nirodents/
RUN pip install --no-cache-dir -e .[all] && \
    rm -rf $HOME/.cache/pip

COPY docker/files/nipype.cfg /home/nirodents/.nipype/nipype.cfg

# Cleanup and ensure perms.
RUN rm -rf $HOME/.npm $HOME/.conda $HOME/.empty && \
    find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} +

# Final settings
WORKDIR /tmp
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="nirodents" \
      org.label-schema.description="nirodents - NeuroImaging workflows" \
      org.label-schema.url="https://github.com/nipreps/nirodents" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/nirodents" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

ENTRYPOINT ["/opt/conda/bin/artsBrainExtraction"]
