# frozen_string_literal: true

module SharedEnvExtension
  # @private
  def effective_arch
    if Homebrew.args.build_bottle?
      ARGV.bottle_arch || Hardware.oldest_cpu
    else
      :native
    end
  end
end
