#!/bin/bash
# Wait for the ConvNeXt-Small blend arm, then push it straight to Kaggle from the box.
# Uploading from the box avoids the slow home hop that stalled a 700MB scp earlier tonight.
cd /workspace/knee
until grep -q "CNS DONE" cns_blend.log 2>/dev/null; do sleep 30; done
rm -rf /workspace/cnsupb && mkdir -p /workspace/cnsupb && cd /workspace/cnsupb
cp /workspace/knee/raptor_ft_cns384_blend.pt .
cat > dataset-metadata.json <<EOF
{"title":"Knee MRI - ConvNeXt-Small 384 Arm B","id":"dreaddevelopment/raptor-knee-arm-cns384b","licenses":[{"name":"CC0-1.0"}]}
EOF
md5sum *.pt
kaggle datasets create -p /workspace/cnsupb -q
echo "CNS-B UPLOADED $(date -u +%H:%M:%S)"
