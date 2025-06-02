george-alisson.html-preview-vscode@0.2.5
github.copilot@1.326.0
github.copilot-chat@0.27.2
goessner.mdmath@2.7.4
grapecity.gc-excelviewer@4.2.63
ionide.ionide-fsharp@7.26.1
mechatroner.rainbow-csv@3.19.0
ms-azuretools.vscode-containers@2.0.2
ms-azuretools.vscode-docker@2.0.0
ms-dotnettools.csharp@2.76.27
ms-dotnettools.dotnet-interactive-vscode@1.0.6177010
ms-dotnettools.vscode-dotnet-pack@1.0.13
ms-dotnettools.vscode-dotnet-runtime@2.3.5
ms-python.debugpy@2025.8.0
ms-python.pylint@2025.2.0
ms-python.python@2025.6.1
ms-python.vscode-pylance@2025.5.1
ms-toolsai.jupyter@2025.4.1
ms-toolsai.jupyter-keymap@1.1.2
ms-toolsai.jupyter-renderers@1.1.0
ms-toolsai.vscode-jupyter-cell-tags@0.1.9
ms-toolsai.vscode-jupyter-slideshow@0.1.6
ms-vscode-remote.remote-containers@0.413.0
ms-vscode-remote.remote-ssh@0.120.0
ms-vscode-remote.remote-ssh-edit@0.87.0
ms-vscode.remote-explorer@0.5.0
ms-vsliveshare.vsliveshare@1.0.5948
njpwerner.autodocstring@0.6.1
rdebugger.r-debugger@0.5.5
redhat.vscode-xml@0.29.0
reditorsupport.r@2.8.6
reditorsupport.r-syntax@0.1.2
rioj7.vscode-remove-comments@1.14.1
saoudrizwan.claude-dev@3.17.8
tomoki1207.pdf@1.2.2
vscodevim.vim@1.30.1
yzhang.markdown-all-in-one@3.6.3

Create a new type of block/lobe (groups of FC layers, activations, optional softmax). I should be able to use the same class in both HEdgeVAMPNetEstimator and VAMPNetEstimator.  Put this in a new submodule called 'layers'.  

Create a new optimizer class which mimics the use of HedgeVAMPNetsEstimator.update_weights. It should be able to be used in a similar way to 'Adam' or any other PyTorch optimizer.