# typed: false
# frozen_string_literal: true

class Pathname
  # Like regular `rmtree`, except it never ignores errors.
  #
  # This was the default behaviour in Ruby 3.1 and earlier.
  #
  # @api public
  def rmtree(noop: nil, verbose: nil, secure: nil)
    # odeprecated "rmtree", "FileUtils#rm_r"
    FileUtils.rm_r(@path, noop:, verbose:, secure:)
    nil
  end
end
