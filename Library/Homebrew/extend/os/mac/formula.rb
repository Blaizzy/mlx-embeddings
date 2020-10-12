# typed: true
# frozen_string_literal: true

class Formula
  undef on_macos

  def on_macos(&_block)
    raise "No block content defined for on_macos block" unless block_given?

    yield
  end

  class << self
    undef on_macos

    def on_macos(&_block)
      raise "No block content defined for on_macos block" unless block_given?

      yield
    end
  end
end
