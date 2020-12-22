# typed: strict
# frozen_string_literal: true

module Hardware
  extend T::Sig
  sig { params(version: T.nilable(Version)).returns(Symbol) }
  def self.oldest_cpu(version = MacOS.version)
    if CPU.arch == :arm64
      :arm_vortex_tempest
    elsif version >= :big_sur
      :ivybridge
    elsif version >= :mojave
      :nehalem
    else
      generic_oldest_cpu
    end
  end
end
