# Image for the aibom-discovery init container. generate_snapshot.py is
# stdlib-only, so this only needs python3, bash and the tools it shells out to:
# lscpu and lsblk (util-linux), df, awk, sed, grep. nvidia-smi isn't installed:
# on GPU pods the NVIDIA Container Toolkit injects it from the node's driver.
# About 200 MB, against about 3 GB for the pytorch/pytorch image it replaces.
FROM registry.access.redhat.com/ubi9/python-312-minimal@sha256:0ee75328b9fd716773396a1e9f86a9ca8b3cde9818078c43f920d4e556f78c2e
USER 0
RUN microdnf install -y --nodocs util-linux && microdnf clean all
USER 1001
