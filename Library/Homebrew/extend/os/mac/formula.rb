# typed: true
# frozen_string_literal: true

class Formula
  undef on_macos

  def on_macos(&block)
    raise "No block content defined for on_macos block" unless block

    yield
  end

  class << self
    undef on_macos

    def on_macos(&block)
      raise "No block content defined for on_macos block" unless block

      yield
    end
  end
end
