# typed: false
# frozen_string_literal: true

module Homebrew
  # Helper functions available in formula `test` blocks.
  #
  # @api private
  module Assertions
    include Context

    require "minitest"
    require "minitest/assertions"
    include ::Minitest::Assertions

    attr_writer :assertions

    def assertions
      @assertions ||= 0
    end

    # Test::Unit backwards compatibility methods
    {
      assert_include:         :assert_includes,
      assert_path_exist:      :assert_path_exists,
      assert_raise:           :assert_raises,
      assert_throw:           :assert_throws,
      assert_not_empty:       :refute_empty,
      assert_not_equal:       :refute_equal,
      assert_not_in_delta:    :refute_in_delta,
      assert_not_in_epsilon:  :refute_in_epsilon,
      assert_not_include:     :refute_includes,
      assert_not_includes:    :refute_includes,
      assert_not_instance_of: :refute_instance_of,
      assert_not_kind_of:     :refute_kind_of,
      assert_not_match:       :refute_match,
      assert_no_match:        :refute_match,
      assert_not_nil:         :refute_nil,
      assert_not_operator:    :refute_operator,
      assert_path_not_exist:  :refute_path_exists,
      assert_not_predicate:   :refute_predicate,
      assert_not_respond_to:  :refute_respond_to,
      assert_not_same:        :refute_same,
    }.each do |old_method, new_method|
      define_method(old_method) do |*args|
        odisabled old_method, new_method
        send(new_method, *args)
      end
    end

    def assert_true(act, msg = nil)
      odisabled "assert_true", "assert(...) or assert_equal(true, ...)"
      assert_equal(true, act, msg)
    end

    def assert_false(act, msg = nil)
      odisabled "assert_false", "assert(!...) or assert_equal(false, ...)"
      assert_equal(false, act, msg)
    end

    # Returns the output of running cmd, and asserts the exit status.
    # @api public
    def shell_output(cmd, result = 0)
      ohai cmd
      output = `#{cmd}`
      assert_equal result, $CHILD_STATUS.exitstatus
      output
    rescue Minitest::Assertion
      puts output if verbose?
      raise
    end

    # Returns the output of running the cmd with the optional input, and
    # optionally asserts the exit status.
    # @api public
    def pipe_output(cmd, input = nil, result = nil)
      ohai cmd
      output = IO.popen(cmd, "w+") do |pipe|
        pipe.write(input) unless input.nil?
        pipe.close_write
        pipe.read
      end
      assert_equal result, $CHILD_STATUS.exitstatus unless result.nil?
      output
    rescue Minitest::Assertion
      puts output if verbose?
      raise
    end
  end
end
