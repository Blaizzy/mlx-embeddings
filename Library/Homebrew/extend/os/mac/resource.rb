# frozen_string_literal: true

class Resource
  undef on_macos

  def on_macos(&_block)
    yield
  end
end
