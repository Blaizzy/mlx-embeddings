# typed: true
# frozen_string_literal: true

module OnOS
  def on_macos(&block)
    raise "No block content defined for 'on_macos' block" unless block

    yield
  end
end
