# frozen_string_literal: true

require "version"

class PkgVersion
  include Comparable

  RX = /\A(.+?)(?:_(\d+))?\z/.freeze

  attr_reader :version, :revision

  def self.parse(path)
    _, version, revision = *path.match(RX)
    version = Version.create(version)
    new(version, revision.to_i)
  end

  def initialize(version, revision)
    @version = version
    @revision = revision
  end

  def head?
    version.head?
  end

  def to_s
    if revision.positive?
      "#{version}_#{revision}"
    else
      version.to_s
    end
  end
  alias to_str to_s

  def <=>(other)
    return unless other.is_a?(PkgVersion)

    version_comparison = (version <=> other.version)
    return if version_comparison.nil?

    version_comparison.nonzero? || revision <=> other.revision
  end
  alias eql? ==

  def hash
    version.hash ^ revision.hash
  end
end
