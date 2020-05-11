# frozen_string_literal: true

module SharedEnvExtension
  # @private
  def effective_arch
    if Homebrew.args.build_bottle?
      Homebrew.args.bottle_arch || Hardware.oldest_cpu
    else
      :native
    end
  end
end
