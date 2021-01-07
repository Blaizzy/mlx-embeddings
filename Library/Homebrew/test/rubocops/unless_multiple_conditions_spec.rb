# typed: false
# frozen_string_literal: true

require "rubocops/unless_multiple_conditions"

describe RuboCop::Cop::Style::UnlessMultipleConditions do
  subject(:cop) { described_class.new }

  it "reports an offense when using `unless` with multiple `and` conditions" do
    expect_offense <<~RUBY
      unless foo && bar
      ^^^^^^^^^^^^^^^^^ Avoid using `unless` with multiple conditions.
        something
      end
    RUBY

    expect_offense <<~RUBY
      something unless foo && bar
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Avoid using `unless` with multiple conditions.
    RUBY
  end

  it "reports an offense when using `unless` with multiple `or` conditions" do
    expect_offense <<~RUBY
      unless foo || bar
      ^^^^^^^^^^^^^^^^^ Avoid using `unless` with multiple conditions.
        something
      end
    RUBY

    expect_offense <<~RUBY
      something unless foo || bar
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Avoid using `unless` with multiple conditions.
    RUBY
  end

  it "reports no offenses when using `if` with multiple `and` conditions" do
    expect_no_offenses <<~RUBY
      if !foo && !bar
        something
      end
    RUBY

    expect_no_offenses <<~RUBY
      something if !foo && !bar
    RUBY
  end

  it "reports no offenses when using `if` with multiple `or` conditions" do
    expect_no_offenses <<~RUBY
      if !foo || !bar
        something
      end
    RUBY

    expect_no_offenses <<~RUBY
      something if !foo || !bar
    RUBY
  end

  it "reports no offenses when using `unless` with single condition" do
    expect_no_offenses <<~RUBY
      unless foo
        something
      end
    RUBY

    expect_no_offenses <<~RUBY
      something unless foo
    RUBY
  end
end
