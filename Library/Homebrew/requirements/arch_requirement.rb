# typed: true
# frozen_string_literal: true

require "requirement"

# A requirement on a specific architecture.
#
# @api private
class ArchRequirement < Requirement
  fatal true

  attr_reader :arch

  def initialize(tags)
    @arch = tags.shift
    super(tags)
  end

  satisfy(build_env: false) do
    case @arch
    when :x86_64 then Hardware::CPU.intel? && Hardware::CPU.is_64_bit?
    when :arm, :intel, :ppc then Hardware::CPU.type == @arch
    end
  end

  def message
    "The #{@arch} architecture is required for this software."
  end

  def inspect
    "#<#{self.class.name}: arch=#{@arch.to_s.inspect} #{tags.inspect}>"
  end

  def display_s
    "#{@arch} architecture"
  end
end
