# typed: true
# frozen_string_literal: true

module OnOS
  def on_linux(&block)
    raise "No block content defined for 'on_linux' block" unless block

    yield
  end
end
