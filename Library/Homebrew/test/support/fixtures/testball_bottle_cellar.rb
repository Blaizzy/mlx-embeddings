# typed: true
# frozen_string_literal: true

class TestballBottleCellar < Formula
  def initialize(name = "testball_bottle", path = Pathname.new(__FILE__).expand_path, spec = :stable,
                 alias_path: nil, force_bottle: false)
    super
  end

  DSL_PROC = proc do
    url "file://#{TEST_FIXTURE_DIR}/tarballs/testball-0.1.tbz"
    sha256 TESTBALL_SHA256

    bottle do
      root_url "file://#{TEST_FIXTURE_DIR}/bottles"
      sha256 cellar: :any_skip_relocation, Utils::Bottles.tag.to_sym => "8f9aecd233463da6a4ea55f5f88fc5841718c013f3e2a7941350d6130f1dc149"
    end

    cxxstdlib_check :skip
  end.freeze
  private_constant :DSL_PROC

  DSL_PROC.call

  def self.inherited(other)
    super
    other.instance_eval(&DSL_PROC)
  end

  def install
    prefix.install "bin"
    prefix.install "libexec"
  end
end
