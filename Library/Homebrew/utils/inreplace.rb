# frozen_string_literal: true

module Utils
  # Helper functions for replacing text in files in-place.
  #
  # @api private
  module Inreplace
    # Error during replacement.
    class Error < RuntimeError
      def initialize(errors)
        formatted_errors = errors.reduce(+"inreplace failed\n") do |s, (path, errs)|
          s << "#{path}:\n" << errs.map { |e| "  #{e}\n" }.join
        end
        super formatted_errors.freeze
      end
    end

    module_function

    # Sometimes we have to change a bit before we install. Mostly we
    # prefer a patch but if you need the `prefix` of this formula in the
    # patch you have to resort to `inreplace`, because in the patch
    # you don't have access to any var defined by the formula. Only
    # `HOMEBREW_PREFIX` is available in the embedded patch.
    #
    # `inreplace` supports regular expressions:
    # <pre>inreplace "somefile.cfg", /look[for]what?/, "replace by #{bin}/tool"</pre>
    #
    # @api public
    def inreplace(paths, before = nil, after = nil, audit_result = true) # rubocop:disable Style/OptionalBooleanParameter
      errors = {}

      errors["`paths` (first) parameter"] = ["`paths` was empty"] if paths.blank?

      Array(paths).each do |path|
        str = File.open(path, "rb", &:read) || ""
        s = StringInreplaceExtension.new(str)

        if before.nil? && after.nil?
          yield s
        else
          after = after.to_s if after.is_a? Symbol
          s.gsub!(before, after, audit_result)
        end

        errors[path] = s.errors unless s.errors.empty?

        Pathname(path).atomic_write(s.inreplace_string)
      end

      raise Error, errors unless errors.empty?
    end

    def inreplace_pairs(path, replacement_pairs, read_only_run: false, silent: false)
      str = File.open(path, "rb", &:read)
      contents = StringInreplaceExtension.new(str)
      replacement_pairs.each do |old, new|
        ohai "replace #{old.inspect} with #{new.inspect}" unless silent
        unless old
          contents.errors << "No old value for new value #{new}! Did you pass the wrong arguments?"
          next
        end

        contents.gsub!(old, new)
      end
      raise Error, path => contents.errors unless contents.errors.empty?

      Pathname(path).atomic_write(contents.inreplace_string) unless read_only_run
      contents.inreplace_string
    end
  end
end
