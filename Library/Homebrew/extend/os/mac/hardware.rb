# typed: strict
# frozen_string_literal: true

module Hardware
  extend T::Sig
  sig { params(version: T.nilable(Version)).returns(Symbol) }
  def self.oldest_cpu(version = MacOS.version)
    if CPU.arch == :arm64
      :arm_vortex_tempest
    # TODO: this cannot be re-enabled until either Rosetta 2 supports AVX
    # instructions in bottles or Homebrew refuses to run under Rosetta 2 (when
    # ARM support is sufficiently complete):
    #   https://github.com/Homebrew/homebrew-core/issues/67713
    #
    # elsif version >= :big_sur
    #   :ivybridge
    elsif version >= :mojave
      :nehalem
    else
      generic_oldest_cpu
    end
  end
end
