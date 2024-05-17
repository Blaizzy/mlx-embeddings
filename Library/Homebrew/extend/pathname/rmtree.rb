# typed: false
# frozen_string_literal: true

class Pathname
  # Like regular `rmtree`, except it never ignores errors.
  #
  # This was the default behaviour in Ruby 3.1 and earlier.
  #
  # @api public
  def rmtree(noop: nil, verbose: nil, secure: nil)
    # Ideally we'd odeprecate this but probably can't given gems so let's
    # create a RuboCop autocorrect instead soon.
    # This is why monkeypatching is non-ideal (but right solution to get
    # Ruby 3.3 over the line).
    # odeprecated "rmtree", "FileUtils#rm_r"
    FileUtils.rm_r(@path, noop:, verbose:, secure:)
    nil
  end
end
