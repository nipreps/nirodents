# Use Ubuntu 16.04 LTS
FROM ubuntu:xenial-20200114

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /usr/local/etc/.neurodebian.gpg

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

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /usr/local/etc/neurodebian.gpg
# Installing Neurodebian packages (FSL, AFNI, git)
RUN curl -sSL "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /usr/local/etc/neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    afni=16.2.07~dfsg.1-5~nd16.04+1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV AFNI_MODELPATH="/usr/lib/afni/models" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_TTATLAS_DATASET="/usr/share/afni/atlases" \
    AFNI_PLUGINPATH="/usr/lib/afni/plugins"
ENV PATH="/usr/lib/afni/bin:$PATH"

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

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh && \
    bash Miniconda3-4.5.11-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-4.5.11-Linux-x86_64.sh

ENV PATH=/usr/local/miniconda/bin:$PATH \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONNOUSERSITE=1

# Installing precomputed python packages
RUN conda install -y python=3.7.1 \
                     mkl=2018.0.3 \
                     mkl-service \
                     numpy>=1.16.5 \
                     scipy=1.1.0 \
                     scikit-learn>=0.20 \
                     matplotlib=2.2.2 \
                     pandas=0.24 \
                     libxml2=2.9.8 \
                     libxslt=1.1.32 \
                     graphviz=2.40.1 \
                     traits=4.6.0 \
                     pip=19.1 \
                     zlib; sync && \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda clean --all -y; sync && \
    conda clean -tipsy && sync

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    TEMPLATEFLOW_AUTOUPDATE=0

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

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

ENTRYPOINT ["/usr/local/miniconda/bin/artsBrainExtraction"]
