# typed: strict
# frozen_string_literal: true

module OnOS
  def on_macos(&block)
    raise "No block content defined for 'on_macos' block" unless T.unsafe(block)

    yield
  end
end
