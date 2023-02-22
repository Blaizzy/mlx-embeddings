# typed: strict

module Cask::Artifact::Installer::ManualInstaller
  include Kernel
  requires_ancestor { Cask::Artifact::Installer }
end

module Cask::Artifact::Installer::ScriptInstaller
  requires_ancestor { Cask::Artifact::Installer }
  requires_ancestor { Cask::Artifact::AbstractArtifact }
end
