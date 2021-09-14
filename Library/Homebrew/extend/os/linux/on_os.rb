# typed: strict
# frozen_string_literal: true

module OnOS
  def on_linux(&block)
    raise "No block content defined for 'on_linux' block" unless T.unsafe(block)

    yield
  end
end
