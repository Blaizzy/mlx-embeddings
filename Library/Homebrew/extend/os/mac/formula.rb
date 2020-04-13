# frozen_string_literal: true

class Formula
  class << self
    undef on_macos

    def on_macos(&_block)
      yield
    end
  end
end
