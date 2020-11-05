# typed: true
# frozen_string_literal: true

# An adapter for casks to provide dependency information in a formula-like interface.
class CaskDependent
  def initialize(cask)
    @cask = cask
  end

  def name
    @cask.token
  end

  def full_name
    @cask.full_name
  end

  def runtime_dependencies
    recursive_dependencies
  end

  def deps
    @deps ||= begin
      @cask.depends_on.formula.map do |f|
        Dependency.new f
      end
    end
  end

  def requirements
    @requirements ||= begin
      requirements = []
      dsl_reqs = @cask.depends_on

      dsl_reqs.arch&.each do |arch|
        requirements << ArchRequirement.new([:x86_64]) if arch[:bits] == 64
        requirements << ArchRequirement.new([arch[:type]])
      end
      dsl_reqs.cask.each do |cask_ref|
        requirements << Requirement.new([{ cask: cask_ref }])
      end
      requirements << dsl_reqs.macos if dsl_reqs.macos
      requirements << X11Requirement.new if dsl_reqs.x11

      requirements
    end
  end

  def recursive_dependencies(&block)
    Dependency.expand(self, &block)
  end

  def recursive_requirements(&block)
    Requirement.expand(self, &block)
  end

  def any_version_installed?
    @cask.installed?
  end
end
