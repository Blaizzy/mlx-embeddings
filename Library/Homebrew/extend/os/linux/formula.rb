# frozen_string_literal: true

class Formula
  undef shared_library

  def shared_library(name, version = nil)
    "#{name}.so#{"." unless version.nil?}#{version}"
  end

  class << self
    undef on_linux

    def on_linux(&_block)
      yield
    end
  end
end
