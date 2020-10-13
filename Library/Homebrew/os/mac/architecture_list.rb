# typed: false
# frozen_string_literal: true

require "hardware"

module ArchitectureListExtension
  # @private
  def universal?
    intersects_all?(Hardware::CPU::INTEL_32BIT_ARCHS, Hardware::CPU::INTEL_64BIT_ARCHS)
  end

  def as_arch_flags
    map { |a| "-arch #{a}" }.join(" ")
  end

  protected

  def intersects_all?(*set)
    set.all? do |archset|
      archset.any? { |a| include? a }
    end
  end
end
