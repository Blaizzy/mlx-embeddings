# frozen_string_literal: true

class Resource
  undef on_linux

  def on_linux(&_block)
    yield
  end
end
