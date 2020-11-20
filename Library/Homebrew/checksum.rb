# typed: true
# frozen_string_literal: true

# A formula's checksum.
#
# @api private
class Checksum
  extend Forwardable

  attr_reader :hash_type, :hexdigest

  TYPES = [:sha256].freeze

  def initialize(hash_type, hexdigest)
    @hash_type = hash_type
    @hexdigest = hexdigest.downcase
  end

  delegate [:empty?, :to_s, :length, :[]] => :@hexdigest

  def ==(other)
    case other
    when String
      to_s == other.downcase
    when Checksum
      hash_type == other.hash_type && hexdigest == other.hexdigest
    else
      false
    end
  end
end
