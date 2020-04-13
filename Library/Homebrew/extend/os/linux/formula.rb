# frozen_string_literal: true

class Formula
  class << self
    undef on_linux

    def on_linux(&_block)
      yield
    end
  end
end
