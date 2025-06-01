#!/bin/bash

echo "🗑️ Eliminando Argo CD..."

microk8s kubectl delete -f applications/ --ignore-not-found=true
microk8s kubectl delete -n argocd -f https://raw.githubusercontent.com/argoproj-labs/argocd-image-updater/stable/manifests/install.yaml --ignore-not-found=true
microk8s kubectl delete -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml --ignore-not-found=true
microk8s kubectl delete namespace argocd --ignore-not-found=true

echo "✅ Argo CD eliminado completamente"